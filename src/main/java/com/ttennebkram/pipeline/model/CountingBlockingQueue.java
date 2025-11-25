package com.ttennebkram.pipeline.model;

import java.util.Collection;
import java.util.Iterator;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * A BlockingQueue wrapper that counts total items added.
 */
public class CountingBlockingQueue<E> implements BlockingQueue<E> {
    private final BlockingQueue<E> delegate;
    private long totalAdded = 0;

    public CountingBlockingQueue() {
        this.delegate = new LinkedBlockingQueue<>();
    }

    public CountingBlockingQueue(int capacity) {
        this.delegate = new LinkedBlockingQueue<>(capacity);
    }

    public long getTotalAdded() {
        return totalAdded;
    }

    public void setTotalAdded(long count) {
        this.totalAdded = count;
    }

    @Override
    public boolean add(E e) {
        boolean result = delegate.add(e);
        if (result) totalAdded++;
        return result;
    }

    @Override
    public boolean offer(E e) {
        boolean result = delegate.offer(e);
        if (result) totalAdded++;
        return result;
    }

    @Override
    public void put(E e) throws InterruptedException {
        delegate.put(e);
        totalAdded++;
    }

    @Override
    public boolean offer(E e, long timeout, TimeUnit unit) throws InterruptedException {
        boolean result = delegate.offer(e, timeout, unit);
        if (result) totalAdded++;
        return result;
    }

    @Override
    public E take() throws InterruptedException {
        return delegate.take();
    }

    @Override
    public E poll(long timeout, TimeUnit unit) throws InterruptedException {
        return delegate.poll(timeout, unit);
    }

    @Override
    public E poll() {
        return delegate.poll();
    }

    @Override
    public E peek() {
        return delegate.peek();
    }

    @Override
    public int remainingCapacity() {
        return delegate.remainingCapacity();
    }

    @Override
    public boolean remove(Object o) {
        return delegate.remove(o);
    }

    @Override
    public boolean contains(Object o) {
        return delegate.contains(o);
    }

    @Override
    public int drainTo(Collection<? super E> c) {
        return delegate.drainTo(c);
    }

    @Override
    public int drainTo(Collection<? super E> c, int maxElements) {
        return delegate.drainTo(c, maxElements);
    }

    @Override
    public int size() {
        return delegate.size();
    }

    @Override
    public boolean isEmpty() {
        return delegate.isEmpty();
    }

    @Override
    public Iterator<E> iterator() {
        return delegate.iterator();
    }

    @Override
    public Object[] toArray() {
        return delegate.toArray();
    }

    @Override
    public <T> T[] toArray(T[] a) {
        return delegate.toArray(a);
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        return delegate.containsAll(c);
    }

    @Override
    public boolean addAll(Collection<? extends E> c) {
        boolean result = delegate.addAll(c);
        if (result) totalAdded += c.size();
        return result;
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        return delegate.removeAll(c);
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        return delegate.retainAll(c);
    }

    @Override
    public void clear() {
        delegate.clear();
    }

    @Override
    public E element() {
        return delegate.element();
    }

    @Override
    public E remove() {
        return delegate.remove();
    }
}
