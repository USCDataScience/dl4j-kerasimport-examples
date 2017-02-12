package edu.usc.irds.dl.dl4j.examples;

/**
 * A POJO class for returning class id, score and name
 * Note: Name may be optional
 */
class Label implements Comparable<Label> {

    private int id;
    private double score;
    private String label;

    public Label(int id, double score) {
        this.id = id;
        this.score = score;
    }

    public Label(int id, double score, String label) {
        this(id, score);
        this.label = label;
    }

    public int getId() {
        return id;
    }

    public double getScore() {
        return score;
    }

    public String getLabel() {
        return label;
    }

    @Override
    public String toString() {
        return "Label(" + id + String.format(", %.4f", score)
                + (label == null ? "" : ", " + label ) + ")";
    }

    @Override
    public int compareTo(Label o) {
        return Double.compare(score, o.score);
    }
}
